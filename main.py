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
                 labels={'count': 'Number of Events'})
    fig.update_layout(title="Distribution of Event Types over Years")
    return fig


# Function to create animated scatter_geo map
def animated_map(data):
    fig = px.scatter_geo(data,
                         lat='LATITUDE',
                         lon='LONGITUDE',
                         color='SUB_EVENT_TYPE',
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


st.title('Data From The Frontlines: Unveiling the dynamics of conflicts in Ukraine and the Black Sea region')
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

# fatalities animation


def animated_map(data):
    fig = px.scatter_geo(data,
                         lat='LATITUDE',
                         lon='LONGITUDE',
                         color='SUB_EVENT_TYPE',
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


# Display the figures in the Streamlit app
st.title('Event Animation')

animated_geo_fig = animated_map(data)
st.plotly_chart(animated_geo_fig)

# casualities_Jahidul
# Filter rows with 'Explosions/Remote violence' event type
explosions = data[data['EVENT_TYPE'] == 'Explosions/Remote violence']

# Group by 'YEAR' and 'SUB_EVENT_TYPE', and count occurrences for each year and sub-type
explosions_by_year_subtype = explosions.groupby(['YEAR', 'SUB_EVENT_TYPE']).size().unstack(fill_value=0)

# Plot the change of 'Explosions/Remote violence' sub-types over the years
st.title('Change of Explosions/Remote violence types over the Years')
st.bar_chart(explosions_by_year_subtype, use_container_width=True)


# fatalities map
def animated_fatalities_map(data):
    # Filter out data points where fatalities are 0
    data_filtered = data[data['FATALITIES'] > 0]

    fig = px.scatter_geo(data_filtered,
                         lat='LATITUDE',
                         lon='LONGITUDE',
                         color='FATALITIES',  # Color points by fatalities number
                         animation_frame='YEAR',
                         projection="natural earth",
                         title='Animated Map of Fatalities by Year',
                         color_continuous_scale='Reds',  # Adjust color scale
                         )
    fig.update_layout(
        autosize=True,
        height=600,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),
            scope='europe',
            projection_scale=3,
            bgcolor='rgba(0, 0, 0, 0)',  # Set background color to transparent
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Set plot background color to transparent
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Set paper background color to transparent
        font=dict(color='white'),  # Set font color to white
    )
    return fig


# Display the animated map
fig = animated_fatalities_map(data)
st.plotly_chart(fig)
