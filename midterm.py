import streamlit as st

from dcairbnb.style import setup_page

st.set_page_config(page_title="Airbnb in DC", page_icon=":material/home_work:", layout="wide")
setup_page()

nav = st.navigation([
    st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/market.py", title="1 · The Market", icon=":material/apartment:"),
    st.Page("views/demand.py", title="2 · Demand", icon=":material/show_chart:"),
    st.Page("views/place.py", title="3 · Place & Power", icon=":material/map:"),
    st.Page("views/rules.py", title="4 · Licensing", icon=":material/gavel:"),
    st.Page("views/projector.py", title="Revenue Projector", icon=":material/calculate:"),
    st.Page("views/methods.py", title="Methods", icon=":material/info:"),
], position="top")
nav.run()

st.html('<hr class="section-rule">')
st.caption("Data: [Inside Airbnb](https://insideairbnb.com/get-the-data/), Washington, DC.")
