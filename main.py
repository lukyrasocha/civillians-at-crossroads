import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd

from math import radians, cos, sin, asin, sqrt, log


# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv', sep=';')
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    data['LATITUDE'] = data['LATITUDE'].str.replace(',', '.')
    data['LONGITUDE'] = data['LONGITUDE'].str.replace(',', '.')
    return data
def filter(data):
    top_5_subevent_types = data['SUB_EVENT_TYPE'].value_counts().index[:5]
    filtered_data = data[data['SUB_EVENT_TYPE'].isin(top_5_subevent_types)]
    filtered_data = filtered_data.sort_values(by='YEAR')
    return filtered_data

# Altair Visualization: Event Type Distribution
def plot_event_type_distribution(data):
    chart = alt.Chart(data).mark_bar().encode(
        x='count()',
        y=alt.Y('EVENT_TYPE', sort='-x'),
        color='EVENT_TYPE',
        tooltip=['EVENT_TYPE', 'count()']
    ).properties(title='Event Type Distribution')
    return chart

# Yearly animation plot
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

# Animation map with region geo info
def animated_map(data, gdf):
    min_year = data['YEAR'].min()
    # Get all unique sub-event types
    all_sub_event_types = data['SUB_EVENT_TYPE'].unique()
    # Create a DataFrame with dummy rows for each sub-event type for the initial year
    dummy_rows = pd.DataFrame({'YEAR': [min_year] * len(all_sub_event_types),
                            'SUB_EVENT_TYPE': all_sub_event_types,
                            'LATITUDE': [0] * len(all_sub_event_types),  
                            'LONGITUDE': [0] * len(all_sub_event_types), 
                            'EVENT_ID_CNTY': ['dummy'] * len(all_sub_event_types)})
    # Concatenate the dummy rows with the original filtered_data DataFrame
    data = pd.concat([data, dummy_rows], ignore_index=True)

    # Create a choropleth map
    choropleth = go.Figure(go.Choropleth(
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        z=[0] * len(gdf),  # Replace with your data
        hovertext=gdf['name'],  # Add hover text
        hoverinfo='text',  # Show only hover text
        marker_line=dict(width=1, color='red'),  # Set marker line width to 1 to display only the outline
        showscale=False
    ))

    # Create a scatter_geo plot using Plotly Express
    scatter_geo = px.scatter_geo(data,
                                  lat='LATITUDE',
                                  lon='LONGITUDE',
                                  color='SUB_EVENT_TYPE',
                                  animation_frame='YEAR',
                                  animation_group='SUB_EVENT_TYPE',
                                  title='Map of Events by Year and Event Type',
                                   hover_data={'LATITUDE':False, 'LONGITUDE':False, 'YEAR':False, 'SUB_EVENT_TYPE': False}
                                )
    
    # Add the choropleth traces to the scatter_geo plot
    for trace in choropleth.data:
        scatter_geo.add_trace(trace)

    # Update layout for the combined plot
    scatter_geo.update_layout(
        autosize=True,
        height=800,
        geo=dict(
            center=dict(lat=47.2, lon=31.1),  # You can adjust these values to focus on your region of interest
            scope='europe',  # Change this as necessary
            projection_scale=5.5  # Adjust the scale for a better view
        )
    )
  
    return scatter_geo


data = load_data()
filtered = filter(data)
gdf = gpd.read_file('region.geojson')

st.title('Echoes of Conflict: Look at Turbulence in the Black Sea Region caused by the Russian-Ukrainian War')
st.write("This dashboard displays visualizations of events in the Black Sea region.")

# Display Altair chart
event_type_chart = plot_event_type_distribution(data)
st.altair_chart(event_type_chart, use_container_width=True)

# Display the figures in the Streamlit app
st.title('Event Animation')


animated_bar_fig = animate_yearly_event(data)
st.plotly_chart(animated_bar_fig)

animated_geo_fig = animated_map(filtered, gdf)
st.plotly_chart(animated_geo_fig)



# python3 -m streamlit run main.py     