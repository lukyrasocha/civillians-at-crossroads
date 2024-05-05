import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import seaborn as sns

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
                 labels={'count': 'Number of Events'})
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
    scatter_geo = px.scatter_geo(
        data, lat='LATITUDE', lon='LONGITUDE', color='SUB_EVENT_TYPE', animation_frame='YEAR',
        animation_group='SUB_EVENT_TYPE', title='Map of Events by Year and Event Type',
        hover_data={'LATITUDE': False, 'LONGITUDE': False, 'YEAR': False, 'SUB_EVENT_TYPE': False})

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

# Extract month and year from 'EVENT_DATE'
data['MONTH'] = data['EVENT_DATE'].dt.month
data['YEAR'] = data['EVENT_DATE'].dt.year

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


st.title('Civilians at the Crossroads: The Human Cost of Conflict in the Black Sea Region')

# st.markdown("## Geopolitical Significance of the Black Sea Region")

st.markdown("The Black Sea region, a pivotal area due to its strategic geopolitical location and rich history, has become an arena for modern conflict and power struggles, particularly impacting Ukraine. In recent years, this area has witnessed an escalation in political violence, deeply affecting the lives of countless civilians.")

st.markdown("This narrative aims to shed light on the human cost of these conflicts. By delving into data on various types of events such as battles, violent demonstrations, and other forms of political violence, we seek to understand not just the frequency and types of these events, but their profound impact on civilian populations. From fatalities and injuries to displacement, the ripple effects of these conflicts reach far beyond the immediate areas of unrest, affecting aspects of daily life, economic stability, and regional security.")

st.markdown('## Event Type Distribution')

# Display Altair chart
event_type_chart = plot_event_type_distribution(data)
st.altair_chart(event_type_chart, use_container_width=True)

# Display the figures in the Streamlit app
st.markdown('## Event Animation')


animated_bar_fig = animate_yearly_event(data)
st.plotly_chart(animated_bar_fig)

animated_geo_fig = animated_map(filtered, gdf)
st.plotly_chart(animated_geo_fig)

# casualities_Jahidul
# Filter rows with 'Explosions/Remote violence' event type
explosions = data[data['EVENT_TYPE'] == 'Explosions/Remote violence']

# Group by 'YEAR' and 'SUB_EVENT_TYPE', and count occurrences for each year and sub-type
explosions_by_year_subtype = explosions.groupby(['YEAR', 'SUB_EVENT_TYPE']).size().unstack(fill_value=0)

# Plot the change of 'Explosions/Remote violence' sub-types over the years
st.markdown('## Change of Explosions/Remote violence types over the Years')
st.bar_chart(explosions_by_year_subtype, use_container_width=True)


# Provide context about the Russia vs Ukraine conflict
st.write("The conflict between Russia and Ukraine witnessed a steady employment of shelling, artillery, and missiles from 2018 to 2021. However, a notable surge in these activities occurred in 2022 and 2023, marked by a significant increase in the frequency and intensity of such attacks.")
st.write("Moreover, the years 2022 and 2023 saw a pronounced rise in the utilization of modern air and drone strikes, suggesting a shift towards more technologically advanced warfare strategies.")
st.write("This trend underscores the evolving nature of the conflict and highlights the increasing reliance on advanced weaponry and aerial capabilities by both parties involved.")


# Display the animated map
fig = animated_fatalities_map(data)
st.plotly_chart(fig)


# fatalities heatmap

# Pivot the data to create a heatmap
heatmap_data = data.pivot_table(index='MONTH', columns='YEAR', values='FATALITIES', aggfunc='sum')

# Create the Streamlit app
st.title('Fatalities Heatmap')

# Plot the heatmap
plt.figure(figsize=(10, 6))
ax = sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='g', linewidths=0.5)

# Get the range of years present in the data
years_range = heatmap_data.columns.tolist()

# Set ticks and labels for the x-axis based on the range of years
ax.set_xticks(range(len(years_range)))
ax.set_xticklabels(years_range, rotation=45)

plt.title('Distribution of Fatalities by Month and Year')
plt.xlabel('Year')
plt.ylabel('Month')

# Display the heatmap in the Streamlit app
st.pyplot(plt.gcf())

st.markdown("## References")
st.markdown("1. [ACLED](https://acleddata.com/)")
st.markdown("2. [A brief history of the importance of Black Sea Region](https://www.csis.org/analysis/geostrategic-importance-black-sea-region-brief-history)")
st.markdown("3. [A security strategy for the Black Sea](https://www.atlanticcouncil.org/in-depth-research-reports/report/a-security-strategy-for-the-black-sea/)")
