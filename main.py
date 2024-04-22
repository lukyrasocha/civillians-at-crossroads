import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt



# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    data['LATITUDE'] = data['LATITUDE'].str.replace(',', '.')
    data['LONGITUDE'] = data['LONGITUDE'].str.replace(',', '.')
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


def animate_yearly_event(data):
    # Ensure your data is sorted by 'YEAR' if the animation sequence matters
    data.sort_values('YEAR', inplace=True)

    event_type_counts = data.groupby(['YEAR', 'EVENT_TYPE']).size().reset_index(name='count')

    # Animated distribution of each individual event type over years
    fig = px.bar(event_type_counts, x='EVENT_TYPE', y='count', color='EVENT_TYPE', 
                animation_frame='YEAR', animation_group='EVENT_TYPE',
                range_y=[0, event_type_counts['count'].max()], 
                labels={'count':'Number of Events'})
    fig.update_layout(title="Distribution of Event Types over Years")
    return fig



# Function to create animated scatter_geo map
def animated_map(data):
    fig = px.scatter_geo(data,
                         lat='LATITUDE',
                         lon='LONGITUDE',
                         color='EVENT_TYPE',
                         hover_name='EVENT_TYPE',
                         hover_data=['FATALITIES', 'NOTES'],
                         animation_frame='YEAR',
                         animation_group='EVENT_TYPE',
                         projection="natural earth",
                         title='Map of Events by Year and Event Type')
    fig.update_layout(
        autosize=True,
        height=600,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),  # You can adjust these values to focus on your region of interest
            scope='europe',  # Change this as necessary
            projection_scale=3  # Adjust the scale for a better view
        )
    )
    return fig

st.title('Echoes of Conflict: Look at Turbulence in the Black Sea Region caused by the Russian-Ukrainian War')
st.write("This dashboard displays visualizations of events in the Black Sea region.")

# Display Altair chart
event_type_chart = plot_event_type_distribution(data)
st.altair_chart(event_type_chart, use_container_width=True)

# Display the figures in the Streamlit app
st.title('Event Animation')


animated_bar_fig = animate_yearly_event(data)
st.plotly_chart(animated_bar_fig)

animated_geo_fig = animated_map(data)
st.plotly_chart(animated_geo_fig)