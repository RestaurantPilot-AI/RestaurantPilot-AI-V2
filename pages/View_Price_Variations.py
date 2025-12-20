"""
Price Variations Dashboard

Shows unit-price variation for menu / supply items for Westman's Bagel & Coffee
(Capitol Hill). The page provides:
- an item-level variation chart (items on the vertical axis, signed unit-price
    change on the horizontal axis),
- a summary table with min/max/absolute/signed changes, and
- item-level price history and invoice listings.

All data is read through the project's `src.storage.database` helpers.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.storage.database import (
    get_price_variations_overview,
    get_item_price_timeseries,
    get_descriptions_for_restaurant,
    get_invoice_line_items_joined,
)

sns.set_theme(style="whitegrid")

# ----- Configuration -----
RESTAURANT_ID = "507f1f77bcf86cd799439011" 


@st.cache_data
def _load_joined(start_date=None, end_date=None):
    return get_invoice_line_items_joined(start_date, end_date, restaurant_ids=[RESTAURANT_ID])


@st.cache_data
def _load_overview(start_date=None, end_date=None, vendor_ids=None, min_occurrences=1):
    return get_price_variations_overview(RESTAURANT_ID, start_date, end_date, vendor_ids, min_occurrences)


@st.cache_data
def _get_descriptions(start_date=None, end_date=None):
    return get_descriptions_for_restaurant(RESTAURANT_ID, start_date, end_date)


st.title("Price Variations")

# Initial joined DF for date range / vendor discovery
base_df = _load_joined()
if base_df is None or base_df.empty:
    st.warning("No invoice/line-item data available for this restaurant.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
valid_dates = base_df["invoice_date"].dropna()
min_date = valid_dates.min().to_pydatetime() if not valid_dates.empty else datetime(2024, 1, 1)
max_date = valid_dates.max().to_pydatetime() if not valid_dates.empty else datetime.now()

date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
start_dt = datetime.combine(date_range[0], datetime.min.time())
end_dt = datetime.combine(date_range[1], datetime.max.time())

vendors = sorted(base_df["vendor"].fillna("Unknown").unique().tolist())
selected_vendor = st.sidebar.selectbox("Vendor", ["All"] + vendors)
vendor_ids = None
if selected_vendor != "All":
    vendor_ids = [selected_vendor]

categories = sorted(base_df["category"].fillna("Uncategorized").unique().tolist())
selected_category = st.sidebar.selectbox("Category (optional)", ["All"] + categories)

min_occ = st.sidebar.slider("Min distinct invoices per item", 1, 10, 1)
top_n = st.sidebar.slider("Top N items to show", 5, 100, 30)

# Load overview dataframe
overview = _load_overview(start_dt, end_dt, vendor_ids, min_occurrences=min_occ)
if overview is None or overview.empty:
    st.warning("No items with price data for selected filters.")
    st.stop()

# Apply category filter
if selected_category != "All":
    overview = overview[overview["category"] == selected_category]

if overview.empty:
    st.warning("No items match the selected filters.")
    st.stop()

# Backwards-compatibility: ensure signed_change columns exist
if "signed_change" not in overview.columns:
    if "first_price" in overview.columns and "last_price" in overview.columns:
        overview["signed_change"] = overview["last_price"] - overview["first_price"]
        def _signed_pct_row(r):
            try:
                if pd.notna(r["first_price"]) and r["first_price"] > 0:
                    return (r["signed_change"] / r["first_price"]) * 100
            except Exception:
                pass
            return float("nan")
        overview["signed_pct_change"] = overview.apply(_signed_pct_row, axis=1)
    else:
        overview["signed_change"] = np.nan
        overview["signed_pct_change"] = np.nan

# Format numbers for display
def _fmt_money(x):
    try:
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return ""
        return f"${f:,.2f}"
    except Exception:
        return ""

def _fmt_pct(x):
    try:
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return ""
        return f"{f:.1f}%"
    except Exception:
        return ""

# Main layout: bar chart then table
st.subheader("Top items by unit-price change")

# Build main chart from the raw joined data (same approach as Category view but
# without applying category filter). This ensures first/last prices are taken
# from actual invoice time order and zeros are included by default.
base_filtered = base_df.copy()
# Apply date filter
base_filtered = base_filtered[(base_filtered["invoice_date"] >= start_dt) & (base_filtered["invoice_date"] <= end_dt)]
# Apply vendor filter if selected
if selected_vendor != "All":
    base_filtered = base_filtered[base_filtered["vendor"] == selected_vendor]

if base_filtered.empty:
    st.info("No line-item data for selected date/vendor filters to build chart.")
else:
    grp_main = base_filtered.sort_values("invoice_date").groupby("item_name").agg(
        first_price=("unit_price", "first"),
        last_price=("unit_price", "last"),
        min_price=("unit_price", "min"),
        max_price=("unit_price", "max"),
        occurrences=("invoice_id", "nunique"),
        vendors=("vendor", lambda s: ", ".join(sorted(s.dropna().unique())))
    ).reset_index()
    grp_main["signed_change"] = grp_main["last_price"] - grp_main["first_price"]
    
    # Filter out items with 0 price change 
    grp_main = grp_main[grp_main["signed_change"] != 0]
    
    # Apply min occurrences threshold
    grp_main = grp_main[grp_main["occurrences"] >= min_occ]
    grp_main = grp_main.assign(sort_key=grp_main["signed_change"].abs()).sort_values("sort_key", ascending=False)
    plot_n = min(top_n, len(grp_main))
    if plot_n == 0:
        st.info("No items meet the occurrence threshold for these filters.")
    else:
        plot_df = grp_main.head(plot_n).copy()
        plt.figure(figsize=(10, max(3, 0.3 * len(plot_df))))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in plot_df["signed_change"]]
        x = plot_df["signed_change"].values
        y_pos = np.arange(len(plot_df))
        bars = plt.barh(y_pos, x, color=colors)
        plt.yticks(y_pos, plot_df["item_name"], fontsize=8)
        plt.xlabel("Unit Price Change ($)")
        plt.ylabel("Item description")
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
        plt.title("Items by Signed Unit Price Change")
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, plot_df["signed_change"]):
            w = bar.get_width()
            offset = 0.02 * (abs(w) if abs(w) > 1 else 1)
            xpos = w + (offset if w >= 0 else -offset)
            ha = "left" if w >= 0 else "right"
            plt.text(xpos, bar.get_y() + bar.get_height() / 2, f"{_fmt_money(val)}", va="center", ha=ha, color="black", fontsize=8)
        import matplotlib.patches as mpatches
        inc = mpatches.Patch(color="#2ca02c", label="Price increase")
        dec = mpatches.Patch(color="#d62728", label="Price decrease")
        plt.legend(handles=[inc, dec], loc="lower right")
        plt.tight_layout()
        st.pyplot(plt.gcf())

        # Summary table sourced from the same grouped dataframe (no category filter)
        display_out = plot_df.copy()
        # Compute percent change where possible
        def _safe_pct(row):
            try:
                if pd.notna(row["first_price"]) and row["first_price"] and row["first_price"] > 0:
                    return (row["signed_change"] / row["first_price"]) * 100
            except Exception:
                pass
            return float("nan")
        display_out["signed_pct_change"] = display_out.apply(_safe_pct, axis=1)
        display_out["Min Price"] = display_out["min_price"].apply(_fmt_money)
        display_out["Max Price"] = display_out["max_price"].apply(_fmt_money)
        # Format Signed fields
        display_out["Price Change"] = display_out["signed_change"].apply(_fmt_money)
        display_out["Price Change%"] = display_out["signed_pct_change"].apply(_fmt_pct)
        display_out = display_out.rename(columns={
            "item_name": "Item",
            "vendors": "Vendors",
            "occurrences": "Occurrences",
        }, errors="ignore")
        display_cols = ["Item", "Min Price", "Max Price", "Price Change", "Price Change%", "Occurrences", "Vendors"]
        display_cols = [c for c in display_cols if c in display_out.columns]
        st.subheader("Summary Table")
        st.dataframe(display_out[display_cols].reset_index(drop=True), width='stretch', hide_index=True)

# Item selection for timeseries
st.markdown("---")
st.subheader("Item price history")
descriptions = _get_descriptions(start_dt, end_dt)
selected_item = st.selectbox("Choose item (pick from restaurant items)", ["-- Select an item --"] + descriptions)

if selected_item and selected_item != "-- Select an item --":
    ts = get_item_price_timeseries(RESTAURANT_ID, selected_item, start_dt, end_dt, vendor_ids=vendor_ids)
    if ts.empty:
        st.warning("No price history found for this item in the selected range/filters.")
    else:
        # Plot time-series colored by vendor (smaller figure)
        plt.figure(figsize=(8, 3))
        sns.lineplot(data=ts, x="date", y="unit_price", hue="vendor", marker="o")
        plt.ylabel("Unit Price")
        plt.xlabel("Date")
        # Adjust y-limits with padding
        ymin = ts["unit_price"].min()
        ymax = ts["unit_price"].max()
        if pd.notna(ymin) and pd.notna(ymax):
            if ymin == ymax:
                pad = max(0.1, abs(ymax) * 0.05)
                plt.ylim(ymin - pad, ymax + pad)
            else:
                pad_low = abs(ymin) * 0.05 if ymin != 0 else 0.05
                pad_high = abs(ymax) * 0.05 if ymax != 0 else 0.05
                plt.ylim(ymin - pad_low, ymax + pad_high)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
        plt.title(f"Price history for: {selected_item}")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

        st.markdown("**Invoices containing this item**")
        display_ts = ts.copy()
        # Deduplicate by invoice id + invoice number + date + unit_price to remove exact duplicates
        display_ts = display_ts.drop_duplicates(subset=["invoice_id", "invoice_number", "date", "unit_price", "vendor"])
        # Format Date and columns, hide invoice_id, remove index
        display_ts = display_ts.rename(columns={"invoice_number": "Invoice Number", "vendor": "Vendor", "unit_price": "Unit Price", "quantity": "Quantity", "category": "Category"})
        display_ts["Date"] = display_ts["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_ts["Unit Price"] = display_ts["Unit Price"].apply(_fmt_money)
        display_ts = display_ts[["Date", "Invoice Number", "Vendor", "Unit Price", "Quantity", "Category"]].reset_index(drop=True)
        # Hide the dataframe index for clarity and display
        try:
            st.dataframe(display_ts.style.hide_index(), width='stretch', hide_index=True)
        except Exception:
            st.dataframe(display_ts, width='stretch', hide_index=True)

# Category-specific variation view
st.markdown("---")
st.subheader("Category view: items and their signed variations")
cat_choice = st.selectbox("Pick category for deeper view", ["All"] + categories, index=0)
if cat_choice and cat_choice != "All":
    cat_df = overview[overview["category"] == cat_choice]
else:
    cat_df = overview.copy()

if cat_df.empty:
    st.info("No items to show in this category with current filters.")
else:
    # Build category-level overview directly from raw joined data so zeros are included
    if cat_choice != "All":
        raw = base_df[base_df["category"] == cat_choice].copy()
    else:
        raw = base_df.copy()

    if raw.empty:
        st.info("No items in this category for the selected filters.")
    else:
        grp = raw.sort_values("invoice_date").groupby("item_name").agg(
            first_price=("unit_price", "first"),
            last_price=("unit_price", "last"),
            min_price=("unit_price", "min"),
            max_price=("unit_price", "max"),
            occurrences=("invoice_id", "nunique"),
            vendors=("vendor", lambda s: ", ".join(sorted(s.dropna().unique())))
        ).reset_index()
        grp["signed_change"] = grp["last_price"] - grp["first_price"]
        grp = grp.assign(sort_key=grp["signed_change"].abs()).sort_values("sort_key", ascending=False)
        plot_n = min(50, len(grp))
        plt.figure(figsize=(10, max(3, 0.25 * plot_n)))
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in grp["signed_change"].head(plot_n)]
        bars = plt.barh(grp["item_name"].head(plot_n), grp["signed_change"].head(plot_n), color=colors)
        plt.xlabel("Signed Unit Price Change ($)")
        plt.ylabel("Item")
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
        plt.title(f"Category — Items by Signed Price Change ({cat_choice})")
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, grp["signed_change"].head(plot_n)):
            w = bar.get_width()
            xpos = w + (0.02 * (1 if w >= 0 else -1) * (abs(w) if abs(w) > 1 else 1))
            ha = "left" if w >= 0 else "right"
            plt.text(xpos, bar.get_y() + bar.get_height()/2, f"{_fmt_money(val)}", va="center", ha=ha, color="black", fontsize=8)
        st.pyplot(plt.gcf())
