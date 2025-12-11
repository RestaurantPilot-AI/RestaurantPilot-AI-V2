## Author : Nithisha
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import numpy as np
from src.storage.database import db, get_vendor_name_by_id

sns.set_theme(style="whitegrid")

# ---------------------------
# Load & Prepare Data
# ---------------------------
@st.cache_data
def load_data():
    """Load invoice and line item data from MongoDB."""
    try:
        # Fetch all invoices from database
        invoices_cursor = db["invoices"].find({})
        invoices_list = list(invoices_cursor)
        
        if not invoices_list:
            st.error("No invoices found in database.")
            return None
        
        # Extract line items from invoices (embedded structure)
        line_items_list = []
        for invoice in invoices_list:
            invoice_id = invoice["_id"]
            embedded_items = invoice.get("line_items", [])
            for item in embedded_items:
                item["invoice_id"] = invoice_id
                line_items_list.append(item)
        
        if not line_items_list:
            st.error("No line items found.")
            return None
        
        # Convert to DataFrames
        invoices = pd.DataFrame(invoices_list)
        line_items = pd.DataFrame(line_items_list)
        
    except Exception as e:
        st.error(f"Could not load data from database: {e}")
        return None

    # Ensure invoice date conversion early
    if "date" in invoices.columns:
        invoices["date"] = pd.to_datetime(invoices["date"], errors="coerce")

    # Merge - handle both ObjectId and string IDs
    # Convert ObjectId to string for merging
    invoices["_id_str"] = invoices["_id"].astype(str)
    line_items["invoice_id_str"] = line_items["invoice_id"].astype(str)
    
    df = pd.merge(
        line_items,
        invoices[["_id_str", "vendor_id", "restaurant_id", "date", "total_amount"]],
        left_on="invoice_id_str",
        right_on="_id_str",
        how="left",
        suffixes=("_line", "_invoice")
    )

    if df.empty:
        st.warning("Merged dataset is empty.")
        return None

    # Ensure numeric columns - handle Decimal128 from MongoDB
    def safe_numeric(val):
        if pd.isna(val):
            return np.nan
        if hasattr(val, 'to_decimal'):  # Decimal128
            return float(val.to_decimal())
        try:
            return float(val)
        except:
            return np.nan
    
    df["line_total"] = df["line_total"].apply(safe_numeric)
    df["unit_price"] = df["unit_price"].apply(safe_numeric)
    df["quantity"] = df["quantity"].apply(safe_numeric)

    # Ensure date column and month period
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Drop rows without dates for time-based charts but keep for non-time charts
        df["month"] = df["date"].dt.to_period("M").astype(str)
    else:
        df["date"] = pd.NaT
        df["month"] = "Unknown"

    # vendor name lookup (robust)
    def safe_vendor(v):
        try:
            if pd.isna(v):
                return "Unknown"
            name = get_vendor_name_by_id(str(v))
            return name if name else "Unknown"
        except Exception:
            return "Unknown"

    df["vendor_name"] = df["vendor_id"].apply(safe_vendor)

    # Ensure category & description
    if "category" not in df.columns or df["category"].isnull().all():
        # create category column frinvoice_id_stription as fallback
        df["category"] = df.get("category", pd.Series(["Uncategorized"] * len(df)))
        # If category missing entirely, set 'Uncategorized' and we'll fallback in charts
        df["category"].fillna("Uncategorized", inplace=True)

    if "description" not in df.columns:
        df["description"] = "Unknown"

    # Ensure invoice id column exists
    if "invoice_id" not in df.columns:
        df["invoice_id"] = df.get("_id_line", np.nan)

    return df

df = load_data()
if df is None:
    st.stop()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.title("Invoice Analysis Dashboard — Graph-focused")

st.sidebar.header("Filters")
min_date, max_date = df["date"].min(), df["date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    # If time data is very sparse, allow user to proceed but warn
    st.sidebar.warning("Date range could not be fully determined; some time charts may be empty.")
    # Provide reasonable defaults
    min_date = df["date"].dropna().min() if not df["date"].dropna().empty else pd.to_datetime("2000-01-01")
    max_date = df["date"].dropna().max() if not df["date"].dropna().empty else pd.to_datetime("2000-01-01")

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM-DD"
)

df = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

vendors = sorted(df["vendor_name"].dropna().unique().tolist())
selected_vendor = st.sidebar.selectbox("Select Vendor", ["All"] + vendors)
df_view = df if selected_vendor == "All" else df[df["vendor_name"] == selected_vendor]

if df_view.empty:
    st.warning("No data for chosen filters.")
    st.stop()

# ---------------------------
# Utility helpers
# ---------------------------
def safe_group_sum(df_in, by, value="line_total"):
    if df_in.empty:
        return pd.Series(dtype=float)
    g = df_in.groupby(by)[value].sum()
    return g.sort_values(ascending=False)

def sort_month_index(series_or_df):
    out = series_or_df.copy()
    try:
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
    except Exception:
        pass
    return out

# ---------------------------
# ALL-VENDORS Graphs (expanded to many)
# ---------------------------

# A: Total Spend Trend (All vendors combined)
def plot_total_spend_trend_all(df_all):
    s = df_all.groupby("month")["line_total"].sum()
    if s.empty or s.dropna().empty:
        st.warning("Not enough time-series spend data to plot Total Spend Trend.")
        return
    s = sort_month_index(s)
    plt.figure(figsize=(10,4))
    plt.plot(s.index, s.values, marker="o")
    plt.title("Total Spend Trend (All Vendors)")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# B: Vendor Contribution Pie / Donut
def plot_vendor_contribution_pie(df_all):
    grouped = df_all.groupby("vendor_name")["line_total"].sum().sort_values(ascending=False)
    if grouped.empty:
        st.warning("No vendor spend to show vendor contribution.")
        return
    top = grouped.head(10)
    others = grouped.iloc[10:].sum()
    if others > 0:
        top["Other"] = others
    
    plt.figure(figsize=(6,6))
    wedges, texts, autotexts = plt.pie(top.values, labels=top.index, autopct="%1.1f%%", startangle=140)
    plt.title("Vendor Contribution Share (Top 10 + Other)")
    # Donut
    centre = Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre)
    st.pyplot(fig)

# C: Top 20 Items Costing the Most (All Vendors)
def plot_top_items_all(df_all, top_n=20):
    grouped = df_all.groupby("description")["line_total"].sum().sort_values(ascending=False).head(top_n)
    if grouped.empty:
        st.warning("No item-level spend data available.")
        return
    plt.figure(figsize=(10,6))
    sns.barplot(x=grouped.values, y=grouped.index, palette="viridis")
    plt.title(f"Top {top_n} Items by Spend (All Vendors)")
    plt.xlabel("Total Spend")
    plt.ylabel("Item Description")
    st.pyplot(plt.gcf())

# D: Category Share Over Time (Stacked Area Chart)
def plot_category_share_over_time(df_all, top_n=8):
    pivot = df_all.groupby(["month","category"])["line_total"].sum().unstack(fill_value=0)
    if pivot.empty:
        st.warning("Not enough data for category share over time.")
        return
    # Keep top categories by total spend across period, group rest into 'Other'
    totals = pivot.sum().sort_values(ascending=False)
    top_cats = totals.head(top_n).index
    small_cats = [c for c in pivot.columns if c not in top_cats]
    pivot_reduced = pivot[top_cats].copy()
    if small_cats:
        pivot_reduced["Other"] = pivot[small_cats].sum(axis=1)
    pivot_reduced = sort_month_index(pivot_reduced)
    plt.figure(figsize=(12,6))
    pivot_reduced.plot(kind="area", stacked=True, alpha=0.85, figsize=(12,6))
    plt.title("Category Share Over Time (Stacked Area)")
    plt.xlabel("Month")
    plt.ylabel("Spend")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# E: Restaurant Spend Ranking (All Vendors)
def plot_restaurant_ranking_all(df_all, top_n=15):
    grouped = df_all.groupby("restaurant_id")["line_total"].sum().sort_values(ascending=False).head(top_n)
    if grouped.empty:
        st.warning("No restaurant spend data.")
        return
    plt.figure(figsize=(8,6))
    sns.barplot(x=grouped.values, y=[str(x) for x in grouped.index], palette="coolwarm")
    plt.title(f"Top {top_n} Restaurants by Spend (All Vendors)")
    plt.xlabel("Total Spend")
    plt.ylabel("Restaurant ID")
    st.pyplot(plt.gcf())

# F: Price Inflation Trend (Average Unit Price Over Time)
def plot_price_inflation_trend(df_all):
    if df_all["unit_price"].dropna().empty:
        st.warning("No unit_price data available to show inflation trend.")
        return
    avg_price = df_all.groupby("month")["unit_price"].mean()
    if avg_price.empty:
        st.warning("Not enough unit_price time-series data.")
        return
    avg_price = sort_month_index(avg_price)
    plt.figure(figsize=(10,4))
    plt.plot(avg_price.index, avg_price.values, marker="o")
    plt.title("Average Unit Price Over Time (All Vendors)")
    plt.xlabel("Month")
    plt.ylabel("Avg Unit Price")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# G: Boxplot: Item Price Distribution (All Vendors)
def plot_price_distribution_box(df_all):
    if df_all["unit_price"].dropna().shape[0] < 10:
        st.warning("Too few unit_price points for a meaningful boxplot.")
        return
    # Use categories if available, else fall back to top descriptions
    if df_all["category"].nunique() > 1 and df_all["category"].notna().sum() > 10:
        data = df_all[["category", "unit_price"]].dropna()
        top_cats = data["category"].value_counts().head(8).index
        data = data[data["category"].isin(top_cats)]
        plt.figure(figsize=(10,6))
        sns.boxplot(data=data, x="category", y="unit_price")
        plt.title("Unit Price Distribution by Category (Top categories)")
        plt.xlabel("Category")
        plt.ylabel("Unit Price")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
    else:
        # fallback to top item descriptions
        data = df_all[["description", "unit_price"]].dropna()
        top_items = data["description"].value_counts().head(8).index
        data = data[data["description"].isin(top_items)]
        plt.figure(figsize=(10,6))
        sns.boxplot(data=data, x="description", y="unit_price")
        plt.title("Unit Price Distribution by Item (Top items)")
        plt.xlabel("Item")
        plt.ylabel("Unit Price")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

# H: Invoice Count Per Vendor Over Time
def plot_invoice_count_per_vendor(df_all):
    if "invoice_id" not in df_all.columns:
        st.warning("No invoice_id available for invoice-count chart.")
        return
    pivot = df_all.groupby(["month", "vendor_name"])["invoice_id"].nunique().unstack(fill_value=0)
    if pivot.empty:
        st.warning("Not enough invoice-count time-series data.")
        return
    pivot = sort_month_index(pivot)
    plt.figure(figsize=(12,6))
    # Show top vendors by total invoice count
    vendor_totals = pivot.sum().sort_values(ascending=False).head(10).index
    pivot_top = pivot[vendor_totals]
    pivot_top.plot(kind="line", marker="o", figsize=(12,6))
    plt.title("Invoice Count Over Time (Top Vendors)")
    plt.xlabel("Month")
    plt.ylabel("Invoice Count")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# ---------------------------
# Vendor-specific (V2) graphs (kept focused)
# ---------------------------

def plot_vendor_monthly(dfv):
    s = dfv.groupby("month")["line_total"].sum()
    if s.empty:
        st.warning("Not enough monthly spend data for vendor.")
        return
    s = sort_month_index(s)
    plt.figure(figsize=(10,4))
    plt.plot(s.index, s.values, marker="o")
    plt.title("Monthly Spend Trend (Vendor)")
    plt.xlabel("Month")
    plt.ylabel("Total Spend")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

def plot_vendor_category_costdriver(dfv):
    grouped = dfv.groupby("category")["line_total"].sum().sort_values(ascending=False)
    # If category is missing or unhelpful, fallback to description
    if grouped.empty or (grouped.index == "Uncategorized").all():
        grouped = dfv.groupby("description")["line_total"].sum().sort_values(ascending=False).head(12)
        title = "Cost Driver — Items (category missing)"
    else:
        title = "Cost Driver — Category (Vendor)"
    if grouped.empty:
        st.warning("No cost-driver data for vendor.")
        return
    plt.figure(figsize=(8,5))
    sns.barplot(x=grouped.values, y=grouped.index, palette="magma")
    plt.title(title)
    plt.xlabel("Total Spend")
    plt.ylabel("Category / Item")
    st.pyplot(plt.gcf())

def plot_vendor_item_costdriver(dfv):
    grouped = dfv.groupby("description")["line_total"].sum().sort_values(ascending=False).head(12)
    if grouped.empty:
        st.warning("No item-level data for vendor.")
        return
    plt.figure(figsize=(10,6))
    sns.barplot(x=grouped.values, y=grouped.index, palette="cubehelix")
    plt.title("Top Items by Spend (Vendor)")
    plt.xlabel("Spend")
    plt.ylabel("Item Description")
    st.pyplot(plt.gcf())

def plot_vendor_category_trend(dfv):
    pivot = dfv.groupby(["month","category"])["line_total"].sum().unstack(fill_value=0)
    if pivot.empty:
        st.warning("Not enough category time data for vendor.")
        return
    pivot = sort_month_index(pivot)
    plt.figure(figsize=(12,6))
    pivot.plot(marker="o", figsize=(12,6))
    plt.title("Category Spend Over Time (Vendor)")
    plt.xlabel("Month")
    plt.ylabel("Spend")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

def plot_top_restaurants_vendor(dfv, top_n=10):
    grouped = dfv.groupby("restaurant_id")["line_total"].sum().sort_values(ascending=False).head(top_n)
    if grouped.empty:
        st.warning("No restaurant-level spend for vendor.")
        return
    plt.figure(figsize=(8,5))
    sns.barplot(x=grouped.values, y=[str(x) for x in grouped.index], palette="coolwarm")
    plt.title("Top Restaurants by Spend (Vendor)")
    plt.xlabel("Total Spend")
    plt.ylabel("Restaurant ID")
    st.pyplot(plt.gcf())

# ---------------------------
# Layout: Show graphs
# ---------------------------
if selected_vendor == "All":
    st.header("All Vendors — Expanded Graphs")

    # Row 1
    st.markdown("### Total Spend Trend")
    plot_total_spend_trend_all(df_view)

    st.markdown("### Vendor Contribution (Top Vendors)")
    plot_vendor_contribution_pie(df_view)

    # Row 2
    st.markdown("### Top Items by Spend (Global)")
    plot_top_items_all(df_view, top_n=20)

    st.markdown("### Category Share Over Time (Stacked Area)")
    plot_category_share_over_time(df_view, top_n=8)

    # Row 3
    st.markdown("### Restaurant Spend Ranking (All Vendors)")
    plot_restaurant_ranking_all(df_view, top_n=15)

    st.markdown("### Average Unit Price Trend (Price Inflation)")
    plot_price_inflation_trend(df_view)

    # Row 4
    st.markdown("### Unit Price Distribution (Boxplot)")
    plot_price_distribution_box(df_view)

    st.markdown("### Invoice Count per Vendor Over Time")
    plot_invoice_count_per_vendor(df_view)

    # Keep category trend as well if desired (now extra)
    st.markdown("### Category Trend (Line, per-category)")
    try:
        pivot = df_view.groupby(["month","category"])["line_total"].sum().unstack(fill_value=0)
        if not pivot.empty:
            pivot = sort_month_index(pivot)
            plt.figure(figsize=(12,6))
            pivot.plot(marker="o", figsize=(12,6))
            plt.title("Category Spend Over Time (All Vendors)")
            plt.xlabel("Month")
            plt.ylabel("Spend")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        else:
            st.warning("Not enough category trend data.")
    except Exception:
        st.warning("Unable to render category trend.")

else:
    st.header(f"Vendor-specific Graphs — {selected_vendor}")

    st.markdown("### Monthly Spend Trend")
    plot_vendor_monthly(df_view)

    st.markdown("### Cost Driver — Category (or item fallback)")
    plot_vendor_category_costdriver(df_view)

    st.markdown("### Top Items (Vendor)")
    plot_vendor_item_costdriver(df_view)

    st.markdown("### Category Trend (Vendor)")
    plot_vendor_category_trend(df_view)

    st.markdown("### Top Restaurants (Vendor)")
    plot_top_restaurants_vendor(df_view)

    # Add two additional vendor-specific supportive graphs (kept focused)
    st.markdown("### Unit Price Distribution (Vendor)")
    # reuse boxplot function but scoped to vendor
    if df_view["unit_price"].dropna().shape[0] >= 5:
        if df_view["category"].nunique() > 1:
            plt.figure(figsize=(10,6))
            top_cats = df_view["category"].value_counts().head(6).index
            sns.boxplot(data=df_view[df_view["category"].isin(top_cats)], x="category", y="unit_price")
            plt.title("Unit Price Distribution by Category (Vendor)")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        else:
            top_items = df_view["description"].value_counts().head(8).index
            plt.figure(figsize=(10,6))
            sns.boxplot(data=df_view[df_view["description"].isin(top_items)], x="description", y="unit_price")
            plt.title("Unit Price Distribution by Item (Vendor)")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
    else:
        st.warning("Not enough unit_price points for vendor unit-price distribution.")

    st.markdown("### Invoice Size Distribution (Vendor)")
    invoices = df_view.groupby("invoice_id")["line_total"].sum().dropna()
    if invoices.shape[0] >= 3:
        plt.figure(figsize=(8,4))
        plt.hist(invoices.values, bins=20)
        plt.title("Distribution of Spend per Invoice (Vendor)")
        plt.xlabel("Invoice Total")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
    else:
        st.warning("Not enough invoice-level data for distribution plot.")
