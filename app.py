import streamlit as st

# Define pages 
upload = st.Page("pages/Upload_Invoices.py", icon='💼')

# Menu pages
upload_menu = st.Page("pages/Upload_Menu.py", icon='📤')
view_menu = st.Page("pages/View_Menu.py", icon='📋')

view_invoices = st.Page("pages/View_Invoices.py", icon='🎓') 

view_price_variations = st.Page("pages/View_Price_Variations.py", icon='📋') # For analysis report
database_controls = st.Page("pages/Database_Controls.py", icon='🧪') # Demo analysis report
dashboard = st.Page("pages/Dashboard.py", icon='📋')


# Group pages
pg = st.navigation({
    "Upload": [upload, upload_menu],
    "View": [view_invoices, view_menu],
    "Analysis": [view_price_variations, dashboard], # Grouped analysis report
    "DB": [database_controls], 

})

# Run the navigation
pg.run()