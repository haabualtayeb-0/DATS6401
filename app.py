import streamlit as st
import plotly.express as px

st.title("My First App")
st.subheader("Simple streamlit app to showcase Tips dataset")

df = px.data.tips()

st.write("Here's a sample of the data:")
st.write(df.head())

st.write("Here's a scatter plot of the data:")
st.scatter_chart(df, x="total_bill", y="tip")

st.write("This scatterplot shows the relationship between the total bill and the tip amount. You can see a positive relationship between both features. Here, as the total bill increases, the tip amount also tends to increase.")


