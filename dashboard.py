import streamlit as st

# Configure the streamlit page
st.set_page_config(
    page_title="Cross-Asset Market Monitor",
    page_icon=":material/trending_up:",
    layout="wide",
)

# Define each page of the dashboard
equities_page = st.Page(
    page="views/equities.py",
    title="Stocks",
    icon=":material/table_chart_view:",
    default=True,
)

bonds_page = st.Page(
    page="views/bonds.py",
    title="Bonds",
    icon=":material/stacked_line_chart:",
)

commodities_page = st.Page(
    page="views/commodities.py",
    title="Commodities",
    icon=":material/oil_barrel:",
)

currencies_page = st.Page(
    page="views/currencies.py",
    title="Currencies",
    icon=":material/euro_symbol:",
)

# Navigation setup
navigator = st.navigation(
    pages = [
        equities_page,
        bonds_page,
        commodities_page,
        currencies_page,
    ]
)

# Run the app
navigator.run()