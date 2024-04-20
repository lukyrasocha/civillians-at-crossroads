import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt



# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    return data

data = load_data()


# Altair Visualization: Event Type Distribution
def plot_event_type_distribution(data):
    chart = alt.Chart(data).mark_bar().encode(
        x='count()',
        y=alt.Y('EVENT_TYPE', sort='-x'),
        color='EVENT_TYPE',
        tooltip=['EVENT_TYPE', 'count()']
    ).properties(title='Event Type Distribution')
    return chart


st.title('Echoes of Conflict: Look at Turbulence in the Black Sea Region caused by the Russian-Ukrainian War')
st.write("This dashboard displays visualizations of events in the Black Sea region.")

# Display Altair chart
event_type_chart = plot_event_type_distribution(data)
st.altair_chart(event_type_chart, use_container_width=True)
