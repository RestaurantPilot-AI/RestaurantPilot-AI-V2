# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import sys
from pathlib import Path
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bson import ObjectId
from src.storage.database import (
    get_all_restaurants,
    get_all_vendors,
    get_invoice_line_items_joined,
    get_sales_data,
    get_spending_by_period,
    get_category_breakdown,
    get_vendor_spending,
    get_top_items_by_spend,
    get_price_variations,
    get_recent_invoices
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Restaurant Cost & Purchasing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- DATA LOADING FROM DATABASE ----------

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_filter_options():
    """Load available restaurants and vendors for filters."""
    try:
        restaurants = get_all_restaurants()
        vendors = get_all_vendors()
        
        # Convert to simpler format for UI
        # Use location_name if available, otherwise use restaurant name
        restaurant_options = {
            (r.get('location_name') or r.get('name', 'Unknown')): str(r['_id']) 
            for r in restaurants
        }
        vendor_options = {
            v['name']: str(v['_id']) 
            for v in vendors
        }
        
        return restaurant_options, vendor_options
    except Exception as e:
        st.error(f"Failed to load filter options: {e}")
        return {}, {}


def load_data_from_db(start_date, end_date, restaurant_ids=None, vendor_ids=None):
    """
    Load invoice and sales data from MongoDB for the specified filters.
    
    Args:
        start_date: Start date (datetime)
        end_date: End date (datetime)
        restaurant_ids: List of ObjectId strings or None
        vendor_ids: List of ObjectId strings or None
    
    Returns:
        invoices_df, sales_df (pandas DataFrames)
    """
    try:
        # Convert string IDs to ObjectId
        # restaurant_oids = [ObjectId(rid) for rid in restaurant_ids] if restaurant_ids else None
        # vendor_oids = [ObjectId(vid) for vid in vendor_ids] if vendor_ids else None
        
        restaurant_oids = restaurant_ids  # keep as strings for debugging
        vendor_oids = vendor_ids  # keep as strings for debugging
        
        # Load invoice data with line items
        invoices_df = get_invoice_line_items_joined(
            start_date=start_date,
            end_date=end_date,
            restaurant_ids=restaurant_oids,
            vendor_ids=vendor_oids
        )
        
        # Load sales data
        sales_df = get_sales_data(
            start_date=start_date,
            end_date=end_date,
            restaurant_ids=restaurant_oids
        )
        
        return invoices_df, sales_df
        
    except Exception as e:
        print("Failed to load data from database:")
        traceback.print_exc()  # full traceback with file & line numbers

        st.error(f"Failed to load data from database: {e}")

        empty_invoices = pd.DataFrame(columns=[
            "invoice_id", "invoice_number", "invoice_date", "location",
            "vendor", "category", "item_name", "quantity", "unit",
            "unit_price", "line_total"
        ])
        empty_sales = pd.DataFrame(columns=["date", "location", "revenue", "covers"])
        return empty_invoices, empty_sales


# ---------- HELPER FUNCTIONS ----------


def compute_weekly_spend(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["week", "total_spend"])
    weekly = (
        df.set_index("invoice_date")
        .resample("W")["line_total"]
        .sum()
        .reset_index()
    )
    weekly.rename(columns={"invoice_date": "week", "line_total": "total_spend"}, inplace=True)
    return weekly


def compute_monthly_spend(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["month", "total_spend"])
    monthly = (
        df.set_index("invoice_date")
        .resample("M")["line_total"]
        .sum()
        .reset_index()
    )
    monthly["month"] = monthly["invoice_date"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly[["month", "total_spend"]]
    return monthly


def safe_metric(value, fmt="{:,.0f}", default="N/A"):
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)):
        return default
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


# ---------- SIDEBAR FILTERS ----------

st.sidebar.title("Filters")

# Load filter options from database
restaurant_options, vendor_options = load_filter_options()

# Date range: Use a wide default range to show all data
today = datetime.today().date()
# Default to last year of data to capture 2024 invoices
default_start = datetime(2024, 1, 1).date()
default_end = today

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, default_end),
    max_value=today,
    help="Select date range for invoices. Data available from 2024 onwards."
)

# Handle date range input (can be tuple or single date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range if hasattr(date_range, 'year') else today

# Convert to datetime for database queries
start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# Location filter
locations_list = list(restaurant_options.keys())
locations_selected_names = st.sidebar.multiselect(
    "Locations",
    options=locations_list,
    default=locations_list,
)

# Convert selected location names to IDs for database query
locations_selected_ids = [restaurant_options[loc] for loc in locations_selected_names] if locations_selected_names else None

# Vendor filter
vendors_list = list(vendor_options.keys())
vendors_selected_names = st.sidebar.multiselect(
    "Vendors",
    options=vendors_list,
    default=vendors_list,
)

# Convert selected vendor names to IDs for database query
vendors_selected_ids = [vendor_options[v] for v in vendors_selected_names] if vendors_selected_names else None

# Refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# Budget input (overall)
monthly_budget = st.sidebar.number_input(
    "Monthly purchasing budget (overall)",
    min_value=0.0,
    value=40000.0,
    step=1000.0,
    help="Used for Budget vs Actual indicators",
)

price_alert_threshold = st.sidebar.slider(
    "Price alert threshold (% change)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
)


# ---------- LOAD DATA FROM DATABASE ----------

with st.spinner("Loading data from database..."):
    invoices_df, sales_df = load_data_from_db(
        start_datetime,
        end_datetime,
        locations_selected_ids,
        vendors_selected_ids
    )

# Use loaded data directly (already filtered by database query)
filtered_invoices = invoices_df.copy()
filtered_sales = sales_df.copy()

# Show data loading info in sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 Data loaded: {len(invoices_df)} invoice line items")
if not sales_df.empty:
    st.sidebar.caption(f"💰 Sales records: {len(sales_df)}")

# Category filter (client-side since it's for UI only)
if not filtered_invoices.empty:
    categories = sorted(filtered_invoices["category"].unique())
    categories_selected = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories,
    )
    
    if categories_selected and len(categories_selected) < len(categories):
        filtered_invoices = filtered_invoices[filtered_invoices["category"].isin(categories_selected)]
else:
    st.sidebar.warning("⚠️ No invoice data available for selected filters")
    st.sidebar.info("💡 Tip: Try adjusting your date range or location filters")

# Some derived aggregates for KPIs
period_days = max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)

# Handle empty sales data
if not sales_df.empty and pd.notna(sales_df["date"].max()):
    today = sales_df["date"].max().date()
else:
    today = end_date

# Last 7 days / previous 7 days
period_end = pd.to_datetime(end_date)
last7_start = period_end - timedelta(days=6)
prev7_start = last7_start - timedelta(days=7)
prev7_end = last7_start - timedelta(days=1)

last7_inv = filtered_invoices[
    (filtered_invoices["invoice_date"] >= last7_start)
    & (filtered_invoices["invoice_date"] <= period_end)
]
prev7_inv = filtered_invoices[
    (filtered_invoices["invoice_date"] >= prev7_start)
    & (filtered_invoices["invoice_date"] <= prev7_end)
]

last7_spend = last7_inv["line_total"].sum()
prev7_spend = prev7_inv["line_total"].sum()

# Monthly spend current vs previous month (based on filtered range end)
end_month = period_end.to_period("M")
current_month_start = end_month.to_timestamp()
prev_month = (end_month - 1).to_timestamp()
prev_month_end = current_month_start - pd.Timedelta(days=1)
prev_month_start = prev_month

current_month_inv = filtered_invoices[
    (filtered_invoices["invoice_date"] >= current_month_start)
    & (filtered_invoices["invoice_date"] <= period_end)
]
prev_month_inv = filtered_invoices[
    (filtered_invoices["invoice_date"] >= prev_month_start)
    & (filtered_invoices["invoice_date"] <= prev_month_end)
]

current_month_spend = current_month_inv["line_total"].sum()
prev_month_spend = prev_month_inv["line_total"].sum()

# Revenue & covers for cost % / cost per cover
total_purchases = filtered_invoices["line_total"].sum()
total_revenue = filtered_sales["revenue"].sum() if "revenue" in filtered_sales.columns else np.nan
total_covers = filtered_sales["covers"].sum() if "covers" in filtered_sales.columns else np.nan

food_cost_pct = (total_purchases / total_revenue * 100) if total_revenue and total_revenue > 0 else np.nan
cost_per_cover = (total_purchases / total_covers) if total_covers and total_covers > 0 else np.nan


# ---------- HEADER & TABS ----------

st.title("🍽️ Restaurant Cost & Purchasing Dashboard")

tab_overview, tab_vendors, tab_planning, tab_ops = st.tabs(
    ["Overview", "Vendors", "Planning & Seasonality", "Operations"]
)


# ---------- TAB 1: OVERVIEW ----------

with tab_overview:
    st.subheader("Key KPIs")

    col1, col2, col3, col4 = st.columns(4)

    # Food Cost %
    with col1:
        st.metric(
            "Food Cost %",
            value=safe_metric(food_cost_pct, "{:,.1f}%"),
        )

    # Total spend this month vs last month
    month_delta = (
        (current_month_spend - prev_month_spend) / prev_month_spend * 100
        if prev_month_spend > 0
        else None
    )
    with col2:
        st.metric(
            "Total Spend (Current Month)",
            value=safe_metric(current_month_spend, "${:,.0f}"),
            delta=safe_metric(month_delta, "{:+.1f}%") if month_delta is not None else "N/A",
        )

    # Spend last 7 days vs previous 7
    w_delta = (
        (last7_spend - prev7_spend) / prev7_spend * 100
        if prev7_spend > 0
        else None
    )
    with col3:
        st.metric(
            "Spend (Last 7 Days)",
            value=safe_metric(last7_spend, "${:,.0f}"),
            delta=safe_metric(w_delta, "{:+.1f}%") if w_delta is not None else "N/A",
        )

    # Cost per cover
    with col4:
        st.metric(
            "Cost per Cover",
            value=safe_metric(cost_per_cover, "${:,.2f}"),
        )

    st.markdown("---")

    # ---- Weekly spending trend ----
    st.subheader("Weekly Spending Trend")
    weekly_spend = compute_weekly_spend(filtered_invoices)
    if weekly_spend.empty:
        st.info("No invoice data available for the selected filters.")
    else:
        chart = (
            alt.Chart(weekly_spend)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:T", title="Week"),
                y=alt.Y("total_spend:Q", title="Total Spend"),
                tooltip=["week:T", "total_spend:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ---- 2x2 grid: Category, Top Cost Drivers, Price Alerts, Savings ----
    st.subheader("Cost Structure & Opportunities")

    c1, c2 = st.columns(2)

    # Cost by category
    with c1:
        st.caption("Cost by Category")
        if filtered_invoices.empty:
            st.write("No data.")
        else:
            cat_df = (
                filtered_invoices.groupby("category")["line_total"]
                .sum()
                .reset_index()
                .sort_values("line_total", ascending=False)
            )
            cat_chart = (
                alt.Chart(cat_df)
                .mark_arc(innerRadius=50)
                .encode(
                    theta="line_total:Q",
                    color="category:N",
                    tooltip=["category", "line_total"],
                )
                .properties(height=280)
            )
            st.altair_chart(cat_chart, use_container_width=True)
            st.dataframe(cat_df.rename(columns={"line_total": "Total Spend"}))

    # Top cost drivers
    with c2:
        st.caption("Top Cost Drivers (Items by Spend)")
        if filtered_invoices.empty:
            st.write("No data.")
        else:
            top_items = (
                filtered_invoices.groupby("item_name")["line_total"]
                .sum()
                .reset_index()
                .sort_values("line_total", ascending=False)
                .head(15)
            )
            bar_chart = (
                alt.Chart(top_items)
                .mark_bar()
                .encode(
                    x=alt.X("line_total:Q", title="Total Spend"),
                    y=alt.Y("item_name:N", sort="-x", title="Item"),
                    tooltip=["item_name", "line_total"],
                )
                .properties(height=280)
            )
            st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("")

    c3, c4 = st.columns(2)

    # Price Alerts
    with c3:
        st.caption(f"Price Alerts (>{price_alert_threshold}% change)")
        if filtered_invoices.empty:
            st.write("No data.")
        else:
            price_df = (
                filtered_invoices.assign(month=lambda d: d["invoice_date"].dt.to_period("M").dt.to_timestamp())
                .groupby(["item_name", "month"])["unit_price"]
                .mean()
                .reset_index()
            )
            # Compute month-over-month change per item with explicit alignment to avoid length mismatch
            price_df.sort_values(["item_name", "month"], inplace=True)
            price_df["prev_price"] = price_df.groupby("item_name")["unit_price"].shift(1)
            price_df["pct_change"] = np.where(
                price_df["prev_price"] > 0,
                (price_df["unit_price"] - price_df["prev_price"]) / price_df["prev_price"] * 100,
                np.nan,
            )

            alerts = price_df.loc[
                price_df["pct_change"].abs() >= price_alert_threshold
            ].copy()
            alerts["direction"] = np.where(alerts["pct_change"] > 0, "Up", "Down")

            if alerts.empty:
                st.success("No significant price changes detected for the selected period.")
            else:
                alerts_display = alerts.sort_values("pct_change", ascending=False)
                alerts_display["pct_change"] = alerts_display["pct_change"].map(lambda x: f"{x:+.1f}%")
                st.dataframe(
                    alerts_display[
                        ["item_name", "month", "unit_price", "pct_change", "direction"]
                    ].rename(
                        columns={
                            "item_name": "Item",
                            "month": "Month",
                            "unit_price": "Avg Price",
                            "pct_change": "Change",
                        }
                    )
                )

    # Savings Opportunities
    with c4:
        st.caption("Savings Opportunities (Vendor Price Comparison)")
        if filtered_invoices.empty:
            st.write("No data.")
        else:
            grp = (
                filtered_invoices.groupby(["item_name", "vendor"])
                .agg(
                    avg_price=("unit_price", "mean"),
                    total_qty=("quantity", "sum"),
                )
                .reset_index()
            )

            savings_rows = []
            for item, sub in grp.groupby("item_name"):
                sub_sorted = sub.sort_values("avg_price")
                if len(sub_sorted) < 2:
                    continue
                best_vendor_row = sub_sorted.iloc[0]
                best_price = best_vendor_row["avg_price"]
                best_vendor = best_vendor_row["vendor"]

                # For all other vendors, estimate savings if they switched
                for _, row in sub_sorted.iloc[1:].iterrows():
                    price_diff = row["avg_price"] - best_price
                    if price_diff <= 0:
                        continue
                    potential_savings = price_diff * row["total_qty"]
                    savings_rows.append(
                        {
                            "Item": item,
                            "Current Vendor": row["vendor"],
                            "Current Avg Price": row["avg_price"],
                            "Best Vendor": best_vendor,
                            "Best Avg Price": best_price,
                            "Potential Savings": potential_savings,
                        }
                    )

            if not savings_rows:
                st.info("No clear savings opportunities found for the selected filters.")
            else:
                savings_df = pd.DataFrame(savings_rows)
                savings_df.sort_values("Potential Savings", ascending=False, inplace=True)
                st.dataframe(savings_df.head(15))

    st.markdown("---")

    # Budget vs Actual (overall for current month)
    st.subheader("Budget vs Actual (Current Month)")

    if current_month_spend == 0:
        st.info("No spend in the current month for the selected filters.")
    else:
        # Assume current monthly budget from sidebar
        progress = (
            current_month_spend / monthly_budget * 100
            if monthly_budget > 0
            else np.nan
        )
        st.write(f"**Current Month Spend:** {safe_metric(current_month_spend, '${:,.0f}')}")

        if not np.isnan(progress):
            st.progress(min(1.0, progress / 100))
            st.write(f"**Budget Usage:** {progress:.1f}% of {safe_metric(monthly_budget, '${:,.0f}')}")
            if progress <= 80:
                st.success("You are within budget so far this month.")
            elif progress <= 100:
                st.warning("You are approaching your monthly budget.")
            else:
                st.error("You have exceeded your monthly budget.")


# ---------- TAB 2: VENDORS ----------

with tab_vendors:
    st.subheader("Vendor Overview")

    if filtered_invoices.empty:
        st.info("No invoice data for selected filters.")
    else:
        vendor_spend = (
            filtered_invoices.groupby("vendor")["line_total"]
            .sum()
            .reset_index()
            .sort_values("line_total", ascending=False)
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.caption("Spend by Vendor")
            v_chart = (
                alt.Chart(vendor_spend)
                .mark_bar()
                .encode(
                    x=alt.X("line_total:Q", title="Total Spend"),
                    y=alt.Y("vendor:N", sort="-x", title="Vendor"),
                    tooltip=["vendor", "line_total"],
                )
                .properties(height=300)
            )
            st.altair_chart(v_chart, use_container_width=True)

        with col2:
            st.caption("Vendor Stats (Selected Period)")
            n_active_vendors = vendor_spend["vendor"].nunique()
            avg_invoice_amount = (
                filtered_invoices.groupby("invoice_id")["line_total"].sum().mean()
            )
            st.metric("Active Vendors", n_active_vendors)
            st.metric("Average Invoice Amount", safe_metric(avg_invoice_amount, "${:,.0f}"))

        st.markdown("---")

        st.subheader("Item-Level Vendor Comparison")
        st.caption(
            "Compare average prices for the same item across vendors to see where you might save."
        )

        item_select = st.selectbox(
            "Select an item to compare prices across vendors",
            options=sorted(filtered_invoices["item_name"].unique()),
        )

        item_data = filtered_invoices[filtered_invoices["item_name"] == item_select]
        cmp_df = (
            item_data.groupby("vendor")
            .agg(
                avg_price=("unit_price", "mean"),
                total_qty=("quantity", "sum"),
                spend=("line_total", "sum"),
            )
            .reset_index()
        )

        cmp_chart = (
            alt.Chart(cmp_df)
            .mark_bar()
            .encode(
                x=alt.X("avg_price:Q", title="Average Unit Price"),
                y=alt.Y("vendor:N", sort="x", title="Vendor"),
                tooltip=["vendor", "avg_price", "total_qty", "spend"],
            )
            .properties(height=280)
        )
        st.altair_chart(cmp_chart, use_container_width=True)

        st.dataframe(
            cmp_df.rename(
                columns={
                    "vendor": "Vendor",
                    "avg_price": "Average Unit Price",
                    "total_qty": "Total Qty",
                    "spend": "Total Spend",
                }
            )
        )


# ---------- TAB 3: PLANNING & SEASONALITY ----------

with tab_planning:
    st.subheader("Seasonal Cost Analysis & Planning")

    if filtered_invoices.empty:
        st.info("No invoice data for selected filters.")
    else:
        # Seasonal cost for chosen items
        st.caption("Seasonal Cost for Selected Items")

        items = sorted(filtered_invoices["item_name"].unique())
        selected_items = st.multiselect(
            "Select one or more items to see cost over time",
            options=items,
            default=items[:3],
        )

        if selected_items:
            seasonal = (
                filtered_invoices[filtered_invoices["item_name"].isin(selected_items)]
                .assign(month=lambda d: d["invoice_date"].dt.to_period("M").dt.to_timestamp())
                .groupby(["item_name", "month"])["line_total"]
                .sum()
                .reset_index()
            )

            season_chart = (
                alt.Chart(seasonal)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("line_total:Q", title="Total Spend"),
                    color="item_name:N",
                    tooltip=["item_name", "month", "line_total"],
                )
                .properties(height=320)
            )
            st.altair_chart(season_chart, use_container_width=True)
        else:
            st.info("Select at least one item to see seasonal trends.")

        st.markdown("---")

        # Category mix over time (100% stacked bar)
        st.subheader("Category Mix Over Time")

        cat_mix = (
            filtered_invoices.assign(month=lambda d: d["invoice_date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "category"])["line_total"]
            .sum()
            .reset_index()
        )
        if not cat_mix.empty:
            total_per_month = cat_mix.groupby("month")["line_total"].transform("sum")
            cat_mix["share"] = cat_mix["line_total"] / total_per_month * 100

            mix_chart = (
                alt.Chart(cat_mix)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("share:Q", stack="normalize", title="Share of Spend"),
                    color=alt.Color("category:N", title="Category"),
                    tooltip=["month", "category", "share"],
                )
                .properties(height=320)
            )
            st.altair_chart(mix_chart, use_container_width=True)
        else:
            st.info("Not enough data for category mix over time.")

        st.markdown("---")

        # Simple spending forecast using average daily spend
        st.subheader("Simple Spending Forecast")

        # Focus on current month for forecast
        month_mask = (filtered_invoices["invoice_date"] >= current_month_start) & (
            filtered_invoices["invoice_date"] <= period_end
        )
        month_data = filtered_invoices[month_mask]
        days_passed = max(
            1,
            (min(period_end, pd.to_datetime(end_date)) - current_month_start).days + 1,
        )

        if not month_data.empty:
            spent_so_far = month_data["line_total"].sum()
            avg_daily = spent_so_far / days_passed
            days_in_month = (current_month_start + pd.offsets.MonthEnd(0)).day
            projected_month_end = avg_daily * days_in_month

            st.write(
                f"Based on an average daily spend of **{safe_metric(avg_daily, '${:,.0f}')}**, "
                f"your **projected month-end spend** is **{safe_metric(projected_month_end, '${:,.0f}')}**."
            )
            if monthly_budget > 0:
                proj_usage = projected_month_end / monthly_budget * 100
                st.write(
                    f"This is **{proj_usage:.1f}%** of your budget "
                    f"({safe_metric(monthly_budget, '${:,.0f}')})."
                )
                if proj_usage <= 90:
                    st.success("You are on track to stay within budget.")
                elif proj_usage <= 110:
                    st.warning("You are projected to be close to your budget.")
                else:
                    st.error("You are projected to exceed your budget.")
        else:
            st.info("No spend in the current month yet for the selected filters; forecast not available.")


# ---------- TAB 4: OPERATIONS & DELIVERY PATTERNS ----------

with tab_ops:
    st.subheader("Ordering & Delivery Patterns")

    if filtered_invoices.empty:
        st.info("No invoice data for selected filters.")
    else:
        # Deliveries by weekday
        st.caption("Deliveries / Invoices by Weekday")
        inv_count = (
            filtered_invoices.assign(weekday=lambda d: d["invoice_date"].dt.day_name())
            .groupby("weekday")["invoice_id"]
            .nunique()
            .reset_index()
        )
        # Preserve natural weekday ordering
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        inv_count["weekday"] = pd.Categorical(inv_count["weekday"], categories=weekday_order, ordered=True)
        inv_count = inv_count.sort_values("weekday")

        weekday_chart = (
            alt.Chart(inv_count)
            .mark_bar()
            .encode(
                x=alt.X("weekday:N", title="Weekday"),
                y=alt.Y("invoice_id:Q", title="Number of Invoices"),
                tooltip=["weekday", "invoice_id"],
            )
            .properties(height=320)
        )
        st.altair_chart(weekday_chart, use_container_width=True)

        st.markdown("---")

        st.subheader("Recent Invoices Feed")

        # Build invoice summary (one row per invoice)
        invoice_summary = (
            filtered_invoices.groupby(["invoice_id", "invoice_date", "vendor", "location"])["line_total"]
            .sum()
            .reset_index()
            .sort_values("invoice_date", ascending=False)
        )

        st.dataframe(
            invoice_summary.head(20).rename(
                columns={
                    "invoice_id": "Invoice ID",
                    "invoice_date": "Date",
                    "vendor": "Vendor",
                    "location": "Location",
                    "line_total": "Total Amount",
                }
            )
        )

        # Simple location comparison, if multiple locations
        st.markdown("---")
        if "location" in filtered_invoices.columns and filtered_invoices["location"].nunique() > 1:
            st.subheader("Location Comparison")

            loc_spend = (
                filtered_invoices.groupby("location")["line_total"]
                .sum()
                .reset_index()
                .rename(columns={"line_total": "Total Spend"})
            )

            loc_chart = (
                alt.Chart(loc_spend)
                .mark_bar()
                .encode(
                    x=alt.X("location:N", title="Location"),
                    y=alt.Y("Total Spend:Q", title="Total Spend"),
                    tooltip=["location", "Total Spend"],
                )
                .properties(height=320)
            )
            st.altair_chart(loc_chart, use_container_width=True)
        else:
            st.info("Location comparison is available when more than one location is selected/present.")
