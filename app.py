import streamlit as st

# Invoice pages 
upload = st.Page("pages/Upload_Invoices.py", icon='💼')
view_invoices = st.Page("pages/View_Invoices.py", icon='🎓') 

# Menu pages
upload_menu = st.Page("pages/Upload_Menu.py", icon='📤')
view_menu = st.Page("pages/View_Menu.py", icon='📋')

# Buildsheet pages
upload_buildsheet = st.Page("pages/Upload_Buildsheet.py", icon='📤')
view_buildsheet = st.Page("pages/View_Buildsheet.py", icon='📤')

# Analysis pages
view_price_variations = st.Page("pages/View_Price_Variations.py", icon='📋') 
database_controls = st.Page("pages/Database_Controls.py", icon='🧪') 
dashboard = st.Page("pages/Dashboard.py", icon='📋')


# Group pages
pg = st.navigation({
    "Upload": [upload, upload_menu, upload_buildsheet],
    "View": [view_invoices, view_menu, view_buildsheet],
    "Analysis": [view_price_variations, dashboard], # Grouped analysis report
    "DB": [database_controls], 

})

# Run the navigation
pg.run()